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
template <typename T>
class ST1
{
public: // 记得加上public
    int n;
    vector<vector<T>> st;
    ST1(vector<T> &a = {}) : n((int)a.size())
    {
        st = vector<vector<T>>(n + 1, vector<T>(22 + 1));
        build(n, a);
    }
    inline T get(const T &a, const T &b) const { return min(a, b); };
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

template <typename T>
class ST2
{
public: // 记得加上public
    int n;
    vector<vector<T>> st;
    ST2(vector<T> &a = {}) : n((int)a.size())
    {
        st = vector<vector<T>>(n + 1, vector<T>(22 + 1));
        build(n, a);
    }
    inline T get(const T &a, const T &b) const { return max(a, b); };
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
                st[i][p] = max(st[i][p - 1], st[i + (t >> 1)][p - 1]);
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
    int n;
    cin >> n;
    vi a(n + 1);
    ff(i, 1, n) cin >> a[i];
    ST1<int> st1(a);
    ST2<int> st2(a);
    int mn = st1.find(1, n), mx = st2.find(1, n);
    cout << 1;
    ff(i, 2, n - 1)
    {
        int t = st1.find(1, i - 1);
        int u = st2.find(i + 1, n);
        if (a[i] == mn || a[i] == mx || t > a[i] || u < a[i])
        {
            cout << 1;
        }
        else
        {
            cout << 0;
        }
    }
    cout << 1;
    cout << endl;
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