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
const int N = 2e5 + 5;
int fi(int x)
{
    if (x == 0)
        return 0;
    if (x == 1)
        return 1;
    if (x == 2)
        return 2;
    return fi(x - 1) + fi(x - 2);
}
void Murasame()
{
    int n, m;
    cin >> n >> m;
    int x = fi(n), y = x + fi(n - 1);
    priority_queue<int, vector<int>, greater<int>> q;
    while (m--)
    {
        int a, b, c;
        cin >> a >> b >> c;
        while (q.size())
            q.pop();
        q.push(c);
        q.push(a);
        q.push(b);
        bool ok = 0;
        if (q.top() >= x)
        {
            q.pop();
            if (q.top() >= x)
            {
                q.pop();
                if (q.top() >= y)
                {
                    ok = 1;
                }
            }
        }
        cout << ok;
    }
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