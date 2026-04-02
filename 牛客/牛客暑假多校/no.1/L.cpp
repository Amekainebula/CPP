#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
struct Fenwick
{
    // 1 - index
    int n;
    std::vector<int> t;

    Fenwick(int n_ = 0, int a = 0) : t(n_ + 1, a), n(n_) {}

    Fenwick(std::vector<int> a) : t(a.size()), n(a.size() - 1)
    {
        for (int i = 1; i <= n; i++)
        {
            t[i] += a[i];

            int j = i + Lowbit(i);
            if (j <= n)
            {
                t[j] += t[i];
            }
        }
    }

    void operator=(const std::vector<int> &a)
    {
        t.clear();
        t.resize(a.size());
        n = a.size() - 1;

        for (int i = 1; i <= n; i++)
        {
            t[i] += a[i];

            int j = i + Lowbit(i);
            if (j <= n)
            {
                t[j] += t[i];
            }
        }
    }
    void clear()
    {
        n = 0;
        t.clear();
    }

    void resize(int x)
    {
        n = x;
        t.resize(x);
    }

    int Lowbit(int x)
    {
        return x & -x;
    }

    void Add(int x, int k)
    {
        while (x <= n)
        {
            t[x] += k;
            x += Lowbit(x);
        }
    }

    int Prefix(int x)
    {
        int sum = 0;
        while (x > 0)
        {
            sum += t[x];
            x -= Lowbit(x);
        }

        return sum;
    }

    int RangeSum(int l, int r)
    {
        return Prefix(r) - Prefix(l - 1);
    }
};

void Murasame()
{
    int n, q;
    cin >> n >> q;
    vi a(n + 1), b(1 + n + q), c(q + 1), d(q + 1);
    vi sum(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i];
        b[i] = a[i];
    }
    ff(i, 1, q)
    {
        cin >> c[i] >> d[i];
        sum[c[i]] += d[i];
        b[i + n] = sum[c[i]] + a[c[i]];
    }

    sort(all1(b));
    b.erase(unique(all1(b)), b.end());
    auto getid = [&](int x) -> int
    { return lower_bound(all1(b), x) - b.begin(); };

    Fenwick fen(b.size());
    ff(i, 1, n)
    {
        fen.Add(getid(a[i]), 1);
    }
    ff(i, 1, q)
    {
        fen.Add(getid(a[c[i]]), -1);
        a[c[i]] += d[i];
        fen.Add(getid(a[c[i]]), 1);
        
        int l = 0, r = b.size(), ans = 0;
        while (l < r)
        {
            int mid = (l + r + 1) / 2;
            int cnt = fen.Prefix(mid);
            if (n - cnt >= n / 2)
            {
                ans = cnt;
                l = mid;
            }
            else
            {
                r = mid - 1;
            }
        }
        cout << ans << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}