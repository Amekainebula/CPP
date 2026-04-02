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
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
template <typename T>
class ST
{
public: // 记得加上public
    int n;
    vector<vector<T>> st;
    ST(int n = 0, vector<T> &a = {}) : n(n)
    {
        st = vector<vector<T>>(n + 1, vector<T>(22 + 1));
        build(n, a);
    }
    inline T get(const T &a, const T &b) { return min(a, b); };

    void build(int n, vector<T> &a)
    {
        for (int i = 1; i <= n; i++)
            st[i][0] = a[i];
        for (int p = 1, t = 2; t <= n; t <<= 1, p++)
        {
            for (int i = 1; i <= n; i++)
            {
                if (i + t - 1 > n)
                    break;
                st[i][p] = min(st[i][p - 1], st[i + (t >> 1)][p - 1]);
            }
        }
    }
    inline T find(int l, int r)
    {
        int t = (int)log2(r - l + 1);
        return get(st[l][t], st[r - (1 << t) + 1][t]);
    }
};
void Murasame()
{
    int n, q;
    cin >> n >> q;
    vi a(n + 1);
    ff(i, 1, n) cin >> a[i];
    ST<int> st(n, a);
    while (q--)
    {
        int l, r;
        cin >> l >> r;
        cout << st.find(l, r) << " ";
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}