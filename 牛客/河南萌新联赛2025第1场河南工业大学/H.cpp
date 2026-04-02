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
int net[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 右下左上
void Murasame()
{
    int n;
    cin >> n;
    vvi mp(n + 1, vi(n + 1));
    mp[(n + 1) / 2][(n + 1) / 2] = 0;
    int now = 0;
    int cnt = 0;
    int x = (n + 1) / 2, y = (n + 1) / 2;
    while (now < n * n)
    {
        cnt++;
        ff(i, 1, cnt)
        {
            x += net[0][0];
            y += net[0][1];
            mp[x][y] = now + 1;
            now++;
        }
        if(now == n * n)break;
        ff(i, 1, cnt)
        {
            x += net[1][0];
            y += net[1][1];
            mp[x][y] = now + 1;
            now++;
        }
        if(now == n * n)break;
        cnt++;
        ff(i, 1, cnt)
        {
            x += net[2][0];
            y += net[2][1];
            mp[x][y] = now + 1;
            now++;
        }
        if(now == n * n)break;
        ff(i, 1, cnt)
        {
            x += net[3][0];
            y += net[3][1];
            mp[x][y] = now + 1;
            now++;
        }
        if(now == n * n)break;
    }
    ff(i, 1, n) ff(j, 1, n) cout << mp[i][j] << " \n"[j == n];
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